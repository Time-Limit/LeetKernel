#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_29_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<29> mm_instantiator;

public:
  UnalignedM_29_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_29_MMInstantiatorWrapper__;

}  // namespace LLMMM
