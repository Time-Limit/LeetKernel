#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_36_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<36> mm_instantiator;

public:
  UnalignedM_36_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_36_MMInstantiatorWrapper__;

}  // namespace LLMMM
