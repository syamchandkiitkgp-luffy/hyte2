"""
HyTE Observability Module using Arize Phoenix.

This module provides a centralized setup for tracing LLM calls, tool executions,
and agent workflows. It integrates OpenInference instrumentation for LangChain
and OpenAI (which we use for Gemini via custom wrappers if needed, or directly
via the Phoenix SDK if using a supported provider).

Since HyTE uses `gemini_client` (REST-based) and `langgraph`, we need to 
manually instrument some parts or use the LangChain instrumentation if we 
adapt our client to LangChain's interface.

For this implementation, we will:
1. Initialize the Phoenix collector.
2. Provide a decorator `@trace_node` for LangGraph nodes.
3. Provide a decorator `@trace_tool` for internal tools (RAG, CodeGen).
4. Provide a wrapper for `call_gemini` to trace LLM calls as 'LLM' spans.
"""

import os
import functools
import json
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Global Tracer
tracer = None

def setup_observability():
    """
    Initializes Arize Phoenix and OpenTelemetry tracing.
    Should be called at the start of the application (e.g., in app.py).
    """
    global tracer
    
    # 1. Launch Phoenix (starts a local server)
    session = px.launch_app(host="0.0.0.0", port=6006, run_in_thread=True)
    print(f"🚀 Phoenix Observability UI is running at: {session.url}")
    
    # 2. Configure OpenTelemetry to send traces to Phoenix
    endpoint = "http://localhost:6006/v1/traces"
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    trace_api.set_tracer_provider(tracer_provider)
    
    tracer = trace_api.get_tracer(__name__)
    
    # 3. Auto-instrument LangChain (if we used standard LC models)
    # Even if we don't fully use LC models, this helps with LangGraph internals if compatible.
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    
    return session.url

def trace_node(node_name):
    """
    Decorator for LangGraph nodes (Orchestrator, Methodology, etc.)
    Creates a 'CHAIN' span.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            state = None
            if len(args) > 0:
                possible_state = args[0]
                if isinstance(possible_state, dict):
                    state = possible_state
                elif len(args) > 1 and isinstance(args[1], dict):
                    state = args[1]
            
            if not tracer or not state:
                 return func(*args, **kwargs)
            
            with tracer.start_as_current_span(
                name=node_name,
                attributes={
                    "openinference.span.kind": "CHAIN",
                    "input.value": str(state),  # Capture full state as input
                    "current_step": state.get("current_step", "unknown")
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("output.value", str(result)) # Capture full resulting state updates
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                    raise e
        return wrapper
    return decorator

def trace_tool(tool_name):
    """
    Decorator for tool functions (RAG, CodeGen logic).
    Creates a 'TOOL' span.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not tracer:
                return func(*args, **kwargs)
            
            input_str = str(args) + str(kwargs)
            
            with tracer.start_as_current_span(
                name=tool_name,
                attributes={
                    "openinference.span.kind": "TOOL",
                    "tool.name": tool_name,
                    "input.value": input_str
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("output.value", str(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                    raise e
        return wrapper
    return decorator

def trace_llm_call(model_name):
    """
    Decorator for the low-level Gemini client call.
    Creates an 'LLM' span.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(prompt, *args, **kwargs):
            if not tracer:
                return func(prompt, *args, **kwargs)
            
            # Use LLM kind for better visualization in Phoenix
            with tracer.start_as_current_span(
                name=f"LLM Call: {model_name}",
                attributes={
                    "openinference.span.kind": "LLM",
                    "llm.model_name": model_name,
                    "llm.invocation_parameters": str(kwargs.get('generation_config', {})),
                    "input.value": str(prompt)
                }
            ) as span:
                try:
                    # Capture full prompt in the standard attribute
                    span.set_attribute("llm.prompts", [str(prompt)])
                    
                    result = func(prompt, *args, **kwargs)
                    
                    # result is usually the text or a list of answers
                    span.set_attribute("output.value", str(result))
                    
                    # Capture messages in OpenInference format (list of JSON strings)
                    span.set_attribute("llm.input_messages", [json.dumps({"role": "user", "content": str(prompt)})])
                    span.set_attribute("llm.output_messages", [json.dumps({"role": "assistant", "content": str(result)})])
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
                    raise e
        return wrapper
    return decorator
