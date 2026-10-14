#pragma once

extern "C" {
void *makeWindowMetalCompatible(void *window);
// Rendering on a C++ worker thread does not inherit Cocoa's event-loop pool.
void runWithCocoaAutoreleasePool(void (*callback)(void *), void *context);
}
