#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_102_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<102> mm_instantiator;

public:
  UnalignedM_102_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_102_MMInstantiatorWrapper__;

}  // namespace LLMMM
