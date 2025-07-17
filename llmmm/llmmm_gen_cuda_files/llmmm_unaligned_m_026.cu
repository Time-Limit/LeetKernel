#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_26_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<26> mm_instantiator;

public:
  UnalignedM_26_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_26_MMInstantiatorWrapper__;

}  // namespace LLMMM
