#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_64_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<64> mm_instantiator;

public:
  UnalignedM_64_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_64_MMInstantiatorWrapper__;

}  // namespace LLMMM
