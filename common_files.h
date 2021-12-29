#include <stdio.h>
#include <stdlib.h>

size_t read_file(const char* _pathname, char** buf, size_t _bufsiz)
{
    if (*buf == NULL)
    {
        if (_bufsiz > BUFSIZ)
        {
            _bufsiz = BUFSIZ;
            *buf = (char*)realloc(*buf, _bufsiz * sizeof(char));
        }
        else
        {
            *buf = (char*)realloc(*buf, _bufsiz * sizeof(char));
        }
    }

    FILE* fp;
    if ((fp = fopen(_pathname, "r")) == NULL)
    {
        perror("Error reading file");
        free(*buf);
        exit(1);
    }

    size_t bytes_read = fread(*buf, sizeof(char), _bufsiz, fp);

    (*buf)[bytes_read] = '\0';

    fclose(fp);

    return bytes_read + 1;
}
