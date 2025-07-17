#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_17_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<17> mm_instantiator;

public:
  UnalignedM_17_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_17_MMInstantiatorWrapper__;

}  // namespace LLMMM
