#include "llmmm/llmmm.h"

int main()
{
  LLMMM::LLMMM::Instance().tune(2048, 12288);

  LLMMM::LLMMM::Instance().tune(4096, 4096);

  return 0;
}
