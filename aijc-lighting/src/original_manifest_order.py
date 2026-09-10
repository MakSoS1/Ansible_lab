from __future__ import annotations

from collections.abc import Sequence


# Permutation reconstructed from the original test.csv supplied with the task.
# Each value is the row index in lexicographically sorted test IDs corresponding
# to that row of the original manifest.  Keeping indices instead of UUID strings
# makes the provenance explicit and lets us verify the ID set independently.
ORIGINAL_TEST_SORTED_INDEX = [
    54, 87, 90, 178, 74, 149, 169, 155, 29, 255, 91, 186, 269, 200, 227,
    246, 59, 65, 221, 80, 150, 66, 117, 272, 212, 157, 75, 109, 96, 170,
    123, 115, 120, 111, 260, 154, 45, 73, 26, 107, 108, 271, 101, 76, 230,
    63, 217, 262, 211, 265, 156, 93, 68, 233, 205, 159, 284, 263, 35, 268,
    198, 81, 183, 250, 279, 266, 67, 158, 70, 122, 53, 254, 179, 112, 118,
    141, 229, 104, 173, 194, 240, 84, 165, 203, 201, 140, 57, 19, 105, 41,
    77, 248, 51, 89, 16, 34, 55, 152, 168, 235, 58, 30, 86, 188, 71, 290,
    44, 64, 17, 83, 102, 293, 295, 137, 204, 69, 287, 285, 11, 0, 95, 223,
    253, 9, 292, 92, 163, 88, 286, 261, 98, 28, 153, 208, 172, 24, 8, 160,
    164, 224, 20, 177, 192, 25, 239, 46, 139, 32, 182, 171, 166, 288, 162,
    82, 234, 114, 14, 161, 21, 270, 146, 128, 185, 218, 129, 191, 282, 202,
    52, 220, 40, 143, 245, 121, 60, 133, 1, 131, 251, 294, 125, 299, 264,
    242, 275, 226, 291, 148, 126, 136, 61, 213, 181, 252, 3, 219, 280, 97,
    238, 7, 232, 296, 78, 13, 38, 147, 237, 180, 231, 273, 144, 187, 132,
    145, 36, 278, 134, 216, 167, 199, 56, 197, 190, 106, 37, 15, 184, 62,
    247, 110, 100, 6, 249, 214, 2, 72, 99, 142, 47, 297, 18, 176, 33, 259,
    119, 193, 48, 85, 175, 39, 267, 298, 281, 49, 113, 209, 27, 222, 258,
    10, 174, 277, 257, 241, 50, 196, 127, 195, 94, 283, 189, 215, 256, 289,
    276, 43, 12, 4, 22, 236, 79, 244, 135, 228, 23, 116, 151, 207, 210, 31,
    225, 42, 243, 124, 130, 206, 5, 103, 138, 274,
]


def restore_original_test_order(sorted_values: Sequence):
    """Return values in the row order of the original supplied test manifest."""
    if len(sorted_values) != len(ORIGINAL_TEST_SORTED_INDEX):
        raise ValueError(
            f"expected {len(ORIGINAL_TEST_SORTED_INDEX)} sorted test rows, "
            f"got {len(sorted_values)}"
        )
    return [sorted_values[index] for index in ORIGINAL_TEST_SORTED_INDEX]
