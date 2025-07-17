#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_90_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<90> mm_instantiator;

public:
  UnalignedM_90_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_90_MMInstantiatorWrapper__;

}  // namespace LLMMM
