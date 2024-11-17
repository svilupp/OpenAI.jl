module OpenAI

using JSON3
using HTTP
using Dates

abstract type AbstractOpenAIProvider end
Base.@kwdef struct OpenAIProvider <: AbstractOpenAIProvider
    api_key::String = ""
    base_url::String = "https://api.openai.com/v1"
    api_version::String = ""
end
Base.@kwdef struct AzureProvider <: AbstractOpenAIProvider
    api_key::String = ""
    base_url::String = "https://docs-test-001.openai.azure.com/openai/deployments/gpt-35-turbo"
    api_version::String = "2023-03-15-preview"
end

"""
DEFAULT_PROVIDER

Default provider for OpenAI API requests.
"""
const DEFAULT_PROVIDER = let
    api_key = get(ENV, "OPENAI_API_KEY", nothing)
    if api_key === nothing
        OpenAIProvider()
    else
        OpenAIProvider(api_key = api_key)
    end
end

"""
    auth_header(provider::AbstractOpenAIProvider, api_key::AbstractString)

Return the authorization header for the given provider and API key.
"""
auth_header(provider::AbstractOpenAIProvider) = auth_header(provider, provider.api_key)
function auth_header(::OpenAIProvider, api_key::AbstractString)
    isempty(api_key) && throw(ArgumentError("api_key cannot be empty"))
    [
        "Authorization" => "Bearer $api_key",
        "Content-Type" => "application/json",
    ]
end
function auth_header(::AzureProvider, api_key::AbstractString)
    isempty(api_key) && throw(ArgumentError("api_key cannot be empty"))
    [
        "api-key" => api_key,
        "Content-Type" => "application/json",
    ]
end

"""
    build_url(provider::AbstractOpenAIProvider, api::AbstractString)
    
    Return the URL for the given provider and API.
"""
build_url(provider::AbstractOpenAIProvider) = build_url(provider, provider.api)
function build_url(provider::OpenAIProvider, api::String)
    isempty(api) && throw(ArgumentError("api cannot be empty"))
    "$(provider.base_url)/$(api)"
end
function build_url(provider::AzureProvider, api::String)
    isempty(api) && throw(ArgumentError("api cannot be empty"))
    (; base_url, api_version) = provider
    return "$(base_url)/$(api)?api-version=$(api_version)"
end

function build_params(kwargs)
    isempty(kwargs) && return nothing
    buf = IOBuffer()
    JSON3.write(buf, kwargs)
    seekstart(buf)
    return buf
end

function request_body(url, method; input, headers, query, kwargs...)
    input = isnothing(input) ? [] : input

    resp = HTTP.request(method,
        url;
        body = input,
        query = query,
        headers = headers,
        kwargs...)
    return resp, resp.body
end

function request_body_live(url; method, input, headers, streamcallback, kwargs...)
    # Create a StreamCallback based on the provided streamcallback
    callback = if streamcallback isa Function
        # If it's a function, wrap it in a StreamCallback with OpenAIStream flavor
        StreamCallback(
            out = chunk -> streamcallback(String(chunk.data)),
            flavor = OpenAIStream()
        )
    elseif streamcallback isa IO
        # If it's an IO, create a StreamCallback that writes to it
        StreamCallback(
            out = chunk -> write(streamcallback, String(chunk.data)),
            flavor = OpenAIStream()
        )
    elseif streamcallback isa StreamCallback
        # If it's already a StreamCallback, use it as is
        streamcallback
    else
        # Default case, create a basic StreamCallback
        StreamCallback(
            out = chunk -> nothing,
            flavor = OpenAIStream()
        )
    end

    # Use StreamCallbacks.jl's streamed_request!
    body = String(take!(input))
    resp = streamed_request!(callback, url, headers, body; kwargs...)

    # Build the response body from the accumulated chunks
    response_body = build_response_body(callback)

    return resp, response_body
end

function status_error(resp, log = nothing)
    logs = !isnothing(log) ? ": $log" : ""
    error("request status $(resp.message)$logs")
end

function _request(api::AbstractString,
    provider::AbstractOpenAIProvider,
    api_key::AbstractString = provider.api_key;
    method,
    query = nothing,
    http_kwargs,
    streamcallback = nothing,
    additional_headers::AbstractVector = Pair{String, String}[],
    kwargs...)
    # add stream: True to the API call if a stream callback function is passed
    if !isnothing(streamcallback)
        kwargs = (kwargs..., stream = true)
    end

    params = build_params(kwargs)
    url = build_url(provider, api)
    resp, body = let
        # Add whatever other headers we were given
        headers = vcat(auth_header(provider, api_key), additional_headers)

        if isnothing(streamcallback)
            request_body(url,
                method;
                input = params,
                headers = headers,
                query = query,
                http_kwargs...)
        else
            request_body_live(url;
                method,
                input = params,
                headers = headers,
                query = query,
                streamcallback = streamcallback,
                http_kwargs...)
        end
    end
    if resp.status >= 400
        status_error(resp, body)
    else
        return if isnothing(streamcallback)
            OpenAIResponse(resp.status, JSON3.read(body))
        else
            # Handle both StreamCallbacks.jl and legacy streaming responses
            lines = if streamcallback isa StreamCallback
                # StreamCallbacks.jl response is already properly formatted
                String.(filter(!isempty, split(body, "\n")))
            else
                # Legacy streaming response handling
                filter(x -> !isempty(x) && !occursin("[DONE]", x), split(body, "\n"))
            end

            # Parse each line as JSON, handling both formats
            parsed = map(lines) do line
                if startswith(line, "data: ")
                    JSON3.read(line[6:end])
                else
                    JSON3.read(line)
                end
            end

            OpenAIResponse(resp.status, parsed)
        end
    end
end

function openai_request(api::AbstractString,
    api_key::AbstractString;
    method,
    http_kwargs,
    streamcallback = nothing,
    kwargs...)
    global DEFAULT_PROVIDER
    _request(api,
        DEFAULT_PROVIDER,
        api_key;
        method,
        http_kwargs,
        streamcallback = streamcallback,
        kwargs...)
end

function openai_request(api::AbstractString,
    provider::AbstractOpenAIProvider;
    method,
    http_kwargs,
    streamcallback = nothing,
    kwargs...)
    _request(api, provider; method, http_kwargs, streamcallback = streamcallback, kwargs...)
end

struct OpenAIResponse{R}
    status::Int16
    response::R
end

"""
Default model ID for embeddings.
Follows recommendation in OpenAI docs at <https://platform.openai.com/docs/models/embeddings>.
"""
const DEFAULT_EMBEDDING_MODEL_ID = "text-embedding-ada-002"

"""
List models

# Arguments:
- `api_key::String`: OpenAI API key

For additional details, visit <https://platform.openai.com/docs/api-reference/models/list>
"""
function list_models(api_key::String; http_kwargs::NamedTuple = NamedTuple())
    return openai_request("models", api_key; method = "GET", http_kwargs = http_kwargs)
end

"""
Retrieve model

# Arguments:
- `api_key::String`: OpenAI API key
- `model_id::String`: Model id

For additional details, visit <https://platform.openai.com/docs/api-reference/models/retrieve>
"""
function retrieve_model(api_key::String,
    model_id::String;
    http_kwargs::NamedTuple = NamedTuple())
    return openai_request("models/$(model_id)",
        api_key;
        method = "GET",
        http_kwargs = http_kwargs)
end

"""
Create completion

# Arguments:
- `api_key::String`: OpenAI API key
- `model_id::String`: Model id

# Keyword Arguments (check the OpenAI docs for the exhaustive list):
- `temperature::Float64=1.0`: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
- `top_p::Float64=1.0`: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.

For more details about the endpoint and additional arguments, visit <https://platform.openai.com/docs/api-reference/completions>

# HTTP.request keyword arguments:
- `http_kwargs::NamedTuple=NamedTuple()`: Keyword arguments to pass to HTTP.request (e. g., `http_kwargs=(connection_timeout=2,)` to set a connection timeout of 2 seconds).
"""
function create_completion(api_key::String,
    model_id::String;
    http_kwargs::NamedTuple = NamedTuple(),
    kwargs...)
    return openai_request("completions",
        api_key;
        method = "POST",
        http_kwargs = http_kwargs,
        model = model_id,
        kwargs...)
end

"""
Create chat

# Arguments:
- `api_key::String`: OpenAI API key
- `model_id::String`: Model id
- `messages::Vector`: The chat history so far.
- `streamcallback=nothing`: Callback for streaming responses. Can be:
  - A function that takes a String (basic streaming)
  - An IO object to write chunks to
  - A StreamCallback object for advanced streaming control (see StreamCallbacks.jl)

# Keyword Arguments (check the OpenAI docs for the exhaustive list):
- `temperature::Float64=1.0`: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
- `top_p::Float64=1.0`: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.

!!! note
    Do not use `stream=true` option here, instead use the `streamcallback` keyword argument (see the Streaming section below).

For more details about the endpoint and additional arguments, visit <https://platform.openai.com/docs/api-reference/chat>

# HTTP.request keyword arguments:
- `http_kwargs::NamedTuple=NamedTuple()`: Keyword arguments to pass to HTTP.request (e. g., `http_kwargs=(connection_timeout=2,)` to set a connection timeout of 2 seconds).

## Basic Example:

```julia
julia> CC = create_chat("..........", "gpt-4-turbo-preview",
    [Dict("role" => "user", "content"=> "What is the OpenAI mission?")]
);

julia> CC.response.choices[1][:message][:content]
"OpenAI's mission is to ensure artificial general intelligence benefits all of humanity."
```

## Streaming

The package supports three ways to handle streaming responses:

### 1. Basic Streaming (Function Callback)

Pass a function that takes a String argument to process each chunk:

```julia
julia> CC = create_chat(key, "gpt-4-turbo-preview",
    [Dict("role" => "user", "content"=> "Count to 5 slowly")],
    streamcallback = chunk -> println("Received: ", chunk)
);
Received: One
Received: , two
Received: , three
Received: , four
Received: , five
```

### 2. IO Streaming

Stream directly to an IO object:

```julia
julia> output_buffer = IOBuffer()
julia> CC = create_chat(key, "gpt-4-turbo-preview",
    [Dict("role" => "user", "content"=> "Say hello")],
    streamcallback = output_buffer
);
julia> String(take!(output_buffer))
"Hello! How can I help you today?"
```

### 3. Advanced Streaming with StreamCallbacks.jl

For advanced streaming control, use StreamCallbacks.jl's StreamCallback:

```julia
using StreamCallbacks

# Custom chunk processing
callback = StreamCallback(
    # Process each chunk
    out = chunk -> println("Token: ", chunk.content),
    # Use OpenAI-specific stream parsing
    flavor = OpenAIStream()
)

CC = create_chat(key, "gpt-4-turbo-preview",
    [Dict("role" => "user", "content"=> "Count to 3")],
    streamcallback = callback
)
```

For advanced streaming features like custom stream parsing, specialized sinks, or detailed chunk inspection,
refer to the [StreamCallbacks.jl](https://github.com/svilupp/StreamCallbacks.jl) package documentation.
"""
function create_chat(api_key::String,
    model_id::String,
    messages;
    http_kwargs::NamedTuple = NamedTuple(),
    streamcallback = nothing,
    kwargs...)
    return openai_request("chat/completions",
        api_key;
        method = "POST",
        http_kwargs = http_kwargs,
        model = model_id,
        messages = messages,
        streamcallback = streamcallback,
        kwargs...)
end
function create_chat(provider::AbstractOpenAIProvider,
    model_id::String,
    messages;
    http_kwargs::NamedTuple = NamedTuple(),
    streamcallback = nothing,
    kwargs...)
    return openai_request("chat/completions",
        provider;
        method = "POST",
        http_kwargs = http_kwargs,
        model = model_id,
        messages = messages,
        streamcallback = streamcallback,
        kwargs...)
end

"""
Create embeddings

# Arguments:
- `api_key::String`: OpenAI API key
- `input`: The input text to generate the embedding(s) for, as String or array of tokens.
    To get embeddings for multiple inputs in a single request, pass an array of strings
        or array of token arrays. Each input must not exceed 8192 tokens in length.
        - `model_id::String`: Model id. Defaults to $DEFAULT_EMBEDDING_MODEL_ID.

        # Keyword Arguments:
        - `http_kwargs::NamedTuple`: Optional. Keyword arguments to pass to HTTP.request.

        For additional details about the endpoint, visit <https://platform.openai.com/docs/api-reference/embeddings>
        """
function create_embeddings(api_key::String,
    input,
    model_id::String = DEFAULT_EMBEDDING_MODEL_ID;
    http_kwargs::NamedTuple = NamedTuple(),
    kwargs...)
    return openai_request("embeddings",
        api_key;
        method = "POST",
        http_kwargs = http_kwargs,
        model = model_id,
        input,
        kwargs...)
end

"""
Create images

# Arguments:
- `api_key::String`: OpenAI API key
- `prompt`: The input text to generate the image(s) for, as String or array of tokens.
- `n::Integer`: Optional. The number of images to generate. Must be between 1 and 10.
- `size::String`: Optional. Defaults to 1024x1024. The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.

# Keyword Arguments:
- `http_kwargs::NamedTuple`: Optional. Keyword arguments to pass to HTTP.request.
- `response_format::String`: Optional. Defaults to "url". The format of the response. Must be one of "url" or "b64_json".

For additional details about the endpoint, visit <https://platform.openai.com/docs/api-reference/images/create>

# once the request is made,
download like this:
`download(r.response["data"][begin]["url"], "image.png")`
"""
function create_images(api_key::String,
    prompt,
    n::Integer = 1,
    size::String = "256x256";
    http_kwargs::NamedTuple = NamedTuple(),
    kwargs...)
    return openai_request("images/generations",
        api_key;
        method = "POST",
        http_kwargs = http_kwargs,
        prompt,
        kwargs...)
end

include("assistants.jl")

export OpenAIResponse
export list_models
export retrieve_model
export create_chat
export create_completion
export create_embeddings
export create_images

# Assistant exports
export list_assistants
export create_assistant
export get_assistant
export delete_assistant
export modify_assistant

# Thread exports
export create_thread
export retrieve_thread
export delete_thread
export modify_thread

# Message exports
export create_message
export list_messages
export retrieve_message
export delete_message
export modify_message

# Run exports
export create_run
export list_runs
export retrieve_run
export delete_run
export modify_run

end # module
