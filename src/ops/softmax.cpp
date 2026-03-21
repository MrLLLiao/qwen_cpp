// File: softmax.c

#include <stdio.h>
#include <ctype.h>

#define MAX_N 200000
#define bool _Bool
#define true 1
#define false 0

typedef long long ll;

int read_int()
{
    int x = 0, f = 1;
    int ch = getchar();
    while (ch != EOF && !isdigit((unsigned char)ch))
    {
        if (ch == '-')
        {
            f = -1;
        }
        ch = getchar();
    }
    while (ch != EOF && isdigit((unsigned char)ch))
    {
        x = (x << 1) + (x << 3) + (ch ^ 48);
        ch = getchar();
    }
    return x * f;
}

void writeln_int(int x)
{
    if (x < 0)
    {
        putchar('-');
        x = -x;
    }
    char st[60];
    int top = 0;
    do
    {
        st[top++] = (char)(x % 10 + '0');
        x /= 10;
    }
    while (x > 0);
    while (top > 0)
    {
        putchar(st[--top]);
    }
    putchar('\n');
}

int main()
{
    return 0;
}