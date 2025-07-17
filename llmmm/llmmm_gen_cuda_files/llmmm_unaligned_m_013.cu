#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_13_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<13> mm_instantiator;

public:
  UnalignedM_13_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_13_MMInstantiatorWrapper__;

}  // namespace LLMMM
