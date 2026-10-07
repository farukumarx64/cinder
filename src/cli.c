#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        printf("Cinder %s\n", CINDER_VERSION);
        return EXIT_SUCCESS;
    }

    fprintf(stderr, "Usage: cinder --version\n");
    return EXIT_FAILURE;
}
