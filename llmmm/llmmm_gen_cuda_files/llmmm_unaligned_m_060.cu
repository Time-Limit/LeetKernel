#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_60_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<60> mm_instantiator;

public:
  UnalignedM_60_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_60_MMInstantiatorWrapper__;

}  // namespace LLMMM
