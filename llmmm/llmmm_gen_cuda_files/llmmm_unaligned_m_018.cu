#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_18_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<18> mm_instantiator;

public:
  UnalignedM_18_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_18_MMInstantiatorWrapper__;

}  // namespace LLMMM
