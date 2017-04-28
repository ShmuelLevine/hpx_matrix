#include <stdio.h>


int main(int argc, char *argv[])
{
    int pos = 33;
    int neg = -13;

    printf("%d / 8 equals: %d\n", pos, pos/8);
    printf("%d % 8 equals: %d\n", pos, pos%8);
    printf("%d / 8 equals: %d\n", neg, neg/8);
    printf("%d % 8 equals: %d\n", neg, neg%8);

    return 0;
}
