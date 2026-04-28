#pragma once

#include <stddef.h>

void *memset(void *dest, int value, size_t count);
void *memcpy(void *dest, const void *src, size_t count);
void *memmove(void *dest, const void *src, size_t count);

