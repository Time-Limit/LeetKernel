#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_100_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<100> mm_instantiator;

public:
  UnalignedM_100_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_100_MMInstantiatorWrapper__;

}  // namespace LLMMM
