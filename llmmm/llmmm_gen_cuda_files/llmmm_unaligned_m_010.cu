#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_10_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<10> mm_instantiator;

public:
  UnalignedM_10_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_10_MMInstantiatorWrapper__;

}  // namespace LLMMM
