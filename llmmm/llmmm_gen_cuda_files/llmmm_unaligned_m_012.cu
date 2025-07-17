#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_12_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<12> mm_instantiator;

public:
  UnalignedM_12_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_12_MMInstantiatorWrapper__;

}  // namespace LLMMM
