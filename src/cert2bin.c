#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CHUNK_SIZE 65536
#define OUT_BUF_SIZE 8192

// Fast custom input buffering
char buf[CHUNK_SIZE];
int buf_pos = 0;
int buf_len = 0;
FILE *fin;

// Output buffering
uint64_t out_buf[OUT_BUF_SIZE];
int out_buf_len = 0;
FILE *fout;

// Fetches the next character efficiently from the input buffer
int next_char() {
    if (buf_pos >= buf_len) {
        buf_len = fread(buf, 1, CHUNK_SIZE, fin);
        buf_pos = 0;
        if (buf_len == 0) return EOF;
    }
    return buf[buf_pos++];
}

// Converts a hex character to its integer equivalent
int parse_hex(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Flushes the output buffer to the binary file
void flush_out() {
    if (out_buf_len > 0) {
        fwrite(out_buf, sizeof(uint64_t), out_buf_len, fout);
        out_buf_len = 0;
    }
}

// Writes a 64-bit integer to the output buffer
void write_binary(uint64_t val) {
    out_buf[out_buf_len++] = val;
    if (out_buf_len == OUT_BUF_SIZE) {
        flush_out();
    }
}

int main(int argc, char *argv[]) {
    // 1. Validate Command Line Arguments
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_text_file> <output_binary_file> <LIMIT>\n", argv[0]);
        return EXIT_FAILURE;
    }

    long limit = strtol(argv[3], NULL, 10);
    if (limit > 47) {
        fprintf(stderr, "Error: LIMIT cannot be greater than 47 (would overflow 48-bit field).\n");
        return EXIT_FAILURE;
    }

    // 2. Open Files
    fin = fopen(argv[1], "rb");
    if (!fin) {
        perror("Error opening input text file");
        return EXIT_FAILURE;
    }

    fout = fopen(argv[2], "wb");
    if (!fout) {
        perror("Error opening output binary file");
        fclose(fin);
        return EXIT_FAILURE;
    }

    uint64_t expected_line = 1;

    // 3. Process the Text File
    while (1) {
        int c;
        
        // Skip leading whitespace or empty lines
        do { c = next_char(); } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (c == EOF) break;
        
        // Parse line_number
        if (c < '0' || c > '9') {
            fprintf(stderr, "Error: Expected line number at line %llu\n", (unsigned long long)expected_line);
            exit(EXIT_FAILURE);
        }
        
        uint64_t line_num = 0;
        while (c >= '0' && c <= '9') {
            line_num = line_num * 10 + (c - '0');
            c = next_char();
        }
        
        // Ensure strictly consecutive ascending order
        if (line_num != expected_line) {
            fprintf(stderr, "Error: Expected line number %llu, got %llu\n", 
                    (unsigned long long)expected_line, (unsigned long long)line_num);
            exit(EXIT_FAILURE);
        }
        
        // Skip the space after line number
        while (c == ' ' || c == '\t') { c = next_char(); }
        
        if (c == 'u') {
            // Format: line_number u
            write_binary(0xFFFFFFFFFFFFFFFFULL);
            
            // Fast-forward to the end of the line
            while ((c = next_char()) != '\n' && c != EOF);
        } else {
            // Format: line_number d p_bitmask
            if (c < '0' || c > '9') {
                fprintf(stderr, "Error: Expected 'u' or integer 'd' at line %llu\n", (unsigned long long)expected_line);
                exit(EXIT_FAILURE);
            }
            
            uint64_t d_val = 0;
            while (c >= '0' && c <= '9') {
                d_val = d_val * 10 + (c - '0');
                c = next_char();
            }
            
            // Read the space and the 'p' literal
            while (c == ' ' || c == '\t') { c = next_char(); }
            if (c != 'p') {
                fprintf(stderr, "Error: Expected 'p' at line %llu\n", (unsigned long long)expected_line);
                exit(EXIT_FAILURE);
            }
            
            if (d_val > limit) {
                // Ignore the huge hex payload and just write FFs
                write_binary(0xFFFFFFFFFFFFFFFFULL);
                
                // Fast-forward to the end of the line, keeping memory O(1)
                while ((c = next_char()) != '\n' && c != EOF);
            } else {
                // Parse the hexadecimal string
                uint64_t hex_val = 0;
                while (1) {
                    c = next_char();
                    if (c == '\n' || c == '\r' || c == EOF || c == ' ') break;
                    
                    int h = parse_hex(c);
                    if (h < 0) {
                        fprintf(stderr, "Error: Invalid hex character at line %llu\n", (unsigned long long)expected_line);
                        exit(EXIT_FAILURE);
                    }
                    hex_val = (hex_val << 4) | h;
                }
                
                // Validate size in bits: the most significant bit must sit exactly at index d_val
                if ((hex_val >> d_val) != 1) {
                    fprintf(stderr, "Error: Hex value bit size is not exactly d+1 at line %llu\n", (unsigned long long)expected_line);
                    exit(EXIT_FAILURE);
                }
                
                // Pack into the 64-bit integer
                uint64_t out_val = (d_val << 48) | hex_val;
                write_binary(out_val);
            }
        }
        
        expected_line++;
    }

    // 4. Cleanup and flush remaining buffer
    flush_out();
    fclose(fin);
    fclose(fout);

    return EXIT_SUCCESS;
}
