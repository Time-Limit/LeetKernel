#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_75_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<75> mm_instantiator;

public:
  UnalignedM_75_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_75_MMInstantiatorWrapper__;

}  // namespace LLMMM
