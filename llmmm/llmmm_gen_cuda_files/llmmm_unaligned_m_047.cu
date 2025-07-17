#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_47_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<47> mm_instantiator;

public:
  UnalignedM_47_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_47_MMInstantiatorWrapper__;

}  // namespace LLMMM
