#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_1_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<1> mm_instantiator;

public:
  UnalignedM_1_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_1_MMInstantiatorWrapper__;

}  // namespace LLMMM
