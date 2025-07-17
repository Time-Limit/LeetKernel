#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_56_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<56> mm_instantiator;

public:
  UnalignedM_56_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_56_MMInstantiatorWrapper__;

}  // namespace LLMMM
