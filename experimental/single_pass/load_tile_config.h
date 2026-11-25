#ifndef QSFA_RECENT_LOAD_TILE_CONFIG_H
#define QSFA_RECENT_LOAD_TILE_CONFIG_H

// These knobs change only the vector gather/decode micro-tile. The attention
// S1/S2 tiles, packed400 row format and 16-element feature blocks stay fixed.
#ifndef QSFA_RECENT_LOAD_ROWS
#define QSFA_RECENT_LOAD_ROWS 16
#endif
#ifndef QSFA_RECENT_INPUT_ROW_BYTES
#define QSFA_RECENT_INPUT_ROW_BYTES 672
#endif
#ifndef QSFA_RECENT_INPUT_BUFFERS
#define QSFA_RECENT_INPUT_BUFFERS 2
#endif
#ifndef QSFA_RECENT_OUTPUT_BUFFERS
#define QSFA_RECENT_OUTPUT_BUFFERS 2
#endif

#if QSFA_RECENT_LOAD_ROWS != 8 && QSFA_RECENT_LOAD_ROWS != 16 && QSFA_RECENT_LOAD_ROWS != 32
#error "Recent128 load rows must be 8, 16, or 32"
#endif
#if QSFA_RECENT_INPUT_ROW_BYTES != 416 && QSFA_RECENT_INPUT_ROW_BYTES != 672
#error "Recent128 input row allocation must be 416 or 672 bytes"
#endif
#if QSFA_RECENT_INPUT_BUFFERS != 1 && QSFA_RECENT_INPUT_BUFFERS != 2
#error "Recent128 input buffers must be 1 or 2"
#endif
#if QSFA_RECENT_OUTPUT_BUFFERS != 1 && QSFA_RECENT_OUTPUT_BUFFERS != 2
#error "Recent128 output buffers must be 1 or 2"
#endif

// Non-default variants are for the host-validated packed400 MXFP4 route only.
// Its DMA destination stride is 416 bytes; 672 is the original over-allocation.
// Each feature block contains 16 BF16 values. Add one padded row to avoid the
// bank conflict of a power-of-two NZ feature-block stride.
#define QSFA_RECENT_LOAD_NZ_ROWS (QSFA_RECENT_LOAD_ROWS + 1)

static_assert(128 % QSFA_RECENT_LOAD_ROWS == 0,
    "Vector load rows must divide the unchanged 128-key attention tile");
static_assert(QSFA_RECENT_LOAD_ROWS % 2 == 0 && QSFA_RECENT_LOAD_ROWS <= 32,
    "Pair gathers and uint32_t hot-row masks require even rows <= 32");
static_assert(QSFA_RECENT_INPUT_ROW_BYTES >= 416 && QSFA_RECENT_INPUT_ROW_BYTES % 32 == 0,
    "Input allocation must contain each aligned packed400 row");

#endif
