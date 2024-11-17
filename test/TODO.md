# Streaming Tests Todo List

## Basic Functionality
- [ ] Test basic function callback streaming
- [ ] Test IO streaming to buffer
- [ ] Test StreamCallback object streaming
- [ ] Test backward compatibility with existing streamcallback usage

## Toy Streaming Server
- [ ] Create mock OpenAI streaming server
- [ ] Test server with different response patterns
- [ ] Test server with error conditions
- [ ] Test server with [DONE] messages

## StreamCallbacks.jl Integration
- [ ] Test OpenAIStream flavor parsing
- [ ] Test custom chunk processing
- [ ] Test response building from chunks
- [ ] Test different sink types

## Error Handling
- [ ] Test connection errors during streaming
- [ ] Test malformed response handling
- [ ] Test early disconnection recovery
- [ ] Test rate limit handling

## Response Building
- [ ] Test OpenAIResponse construction from streamed chunks
- [ ] Test delta merging in responses
- [ ] Test role and content field handling
- [ ] Test empty response handling

## Edge Cases
- [ ] Test very large responses
- [ ] Test rapid message sequences
- [ ] Test unicode content
- [ ] Test empty messages
