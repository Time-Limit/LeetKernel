#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_105_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<105> mm_instantiator;

public:
  UnalignedM_105_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_105_MMInstantiatorWrapper__;

}  // namespace LLMMM
