#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_77_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<77> mm_instantiator;

public:
  UnalignedM_77_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_77_MMInstantiatorWrapper__;

}  // namespace LLMMM
