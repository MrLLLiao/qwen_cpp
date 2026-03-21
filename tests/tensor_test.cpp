#include "tensor.h"
#include <iostream>

int main()
{
    using namespace std;

    Tensor2D tensor2d(2, 3, 5.0f);

    tensor2d.print();
    putchar('\n');

    size_t rows = tensor2d.rows();
    size_t cols = tensor2d.cols();
    size_t size = tensor2d.size();

    cout << "rows: " << rows << " cols " << cols << " size: " << size << endl;

    cout << "Before change - (1, 1) = " << tensor2d(1, 1) << endl;
    tensor2d.at(1, 1) = 10.0f;
    cout << "After change - (1, 1) = " << tensor2d(1, 1) << endl;

    putchar('\n');
    tensor2d.fill(0.0f);
    tensor2d.print();

    return 0;
}
