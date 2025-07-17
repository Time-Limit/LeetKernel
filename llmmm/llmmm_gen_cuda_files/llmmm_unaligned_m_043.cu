#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_43_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<43> mm_instantiator;

public:
  UnalignedM_43_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_43_MMInstantiatorWrapper__;

}  // namespace LLMMM
