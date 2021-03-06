#pragma once

#include <stdio.h>
#include <stdlib.h>

size_t file_read(const char* _pathname, char** buf, size_t bufsiz)
{
    if (*buf == NULL)
    {
        if (bufsiz > BUFSIZ)
        {
            bufsiz = BUFSIZ;
            *buf = (char*)realloc(*buf, bufsiz * sizeof(char));
        }
        else
        {
            *buf = (char*)realloc(*buf, bufsiz * sizeof(char));
        }
    }

    FILE* fp;
    if ((fp = fopen(_pathname, "r")) == NULL)
    {
        perror("error reading file");
        free(*buf);
        exit(1);
    }

    size_t bytes_read = fread(*buf, sizeof(char), bufsiz - 1, fp);

    (*buf)[bytes_read] = '\0';

    fclose(fp);

    return bytes_read + 1;
}
