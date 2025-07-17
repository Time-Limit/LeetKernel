#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_19_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<19> mm_instantiator;

public:
  UnalignedM_19_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_19_MMInstantiatorWrapper__;

}  // namespace LLMMM
