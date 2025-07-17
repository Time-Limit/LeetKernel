#include "llmmm/llmmm.h"

int main()
{
  LLMMM::LLMMM::Instance().tune(2048, 12288);

  return 0;
}
