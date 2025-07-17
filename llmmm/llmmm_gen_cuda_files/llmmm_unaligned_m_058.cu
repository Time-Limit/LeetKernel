#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_58_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<58> mm_instantiator;

public:
  UnalignedM_58_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_58_MMInstantiatorWrapper__;

}  // namespace LLMMM
