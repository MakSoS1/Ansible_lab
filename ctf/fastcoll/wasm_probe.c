#include <stdint.h>
__attribute__((import_module("env"), import_name("xx_hash")))
extern int host_xxh3(uint32_t ptr, uint32_t len, uint32_t digest_ptr);
__attribute__((aligned(8))) static uint8_t messages[512];
__attribute__((aligned(8))) static uint8_t digests[32];
__attribute__((export_name("xx_hash"))) void xx_hash(void) {
    messages[256] = 1;
    host_xxh3((uint32_t)(uintptr_t)&messages[0], 256, (uint32_t)(uintptr_t)&digests[0]);
    host_xxh3((uint32_t)(uintptr_t)&messages[256], 256, (uint32_t)(uintptr_t)&digests[16]);
}
