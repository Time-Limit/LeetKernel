#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_122_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<122> mm_instantiator;

public:
  UnalignedM_122_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_122_MMInstantiatorWrapper__;

}  // namespace LLMMM
