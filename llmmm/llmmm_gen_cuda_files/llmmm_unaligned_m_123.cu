#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_123_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<123> mm_instantiator;

public:
  UnalignedM_123_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_123_MMInstantiatorWrapper__;

}  // namespace LLMMM
