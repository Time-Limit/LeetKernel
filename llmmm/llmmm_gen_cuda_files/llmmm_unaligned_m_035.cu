#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_35_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<35> mm_instantiator;

public:
  UnalignedM_35_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_35_MMInstantiatorWrapper__;

}  // namespace LLMMM
