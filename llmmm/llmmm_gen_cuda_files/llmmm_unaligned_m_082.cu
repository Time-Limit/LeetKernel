#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_82_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<82> mm_instantiator;

public:
  UnalignedM_82_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_82_MMInstantiatorWrapper__;

}  // namespace LLMMM
