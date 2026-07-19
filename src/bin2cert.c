// This was written by Google Gemini

// Takes an out.found.bin binary file
// as written by coarse_sieve
// and turns it into a text "certificate" file
// in the same format as created by gf2x/apps/factor.cpp
// from the GF2X git repository

// cc -o bin2cert bin2cert.c
// ./bin2cert out.found.bin > out.found.cert

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define BUFFER_SIZE 8192 // Read 8192 records (64KB) at a time for efficiency

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <binary_file>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    uint64_t buffer[BUFFER_SIZE];
    size_t records_read;
    uint64_t line_num = 1;

    // Read the binary file in chunks
    while ((records_read = fread(buffer, sizeof(uint64_t), BUFFER_SIZE, file)) > 0) {
        for (size_t i = 0; i < records_read; i++) {
            uint64_t record = buffer[i];

            // Handle the all-FFs special case ('u')
            if (record == 0xFFFFFFFFFFFFFFFFULL) {
                printf("%llu u\n", (unsigned long long)line_num);
            } else {
                // Extract the upper 16 bits (val2) and lower 48 bits (val1)
                uint32_t val2 = (record >> 48) & 0xFFFF;
                uint64_t val1 = record & 0x0000FFFFFFFFFFFFULL;
                
                printf("%llu %u p%llx\n", (unsigned long long)line_num, val2, (unsigned long long)val1);
            }
            line_num++;
        }
    }

    if (ferror(file)) {
        fprintf(stderr, "Error reading from file.\n");
        fclose(file);
        return 1;
    }

    fclose(file);
    return 0;
}
