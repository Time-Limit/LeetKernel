#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_106_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<106> mm_instantiator;

public:
  UnalignedM_106_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_106_MMInstantiatorWrapper__;

}  // namespace LLMMM
