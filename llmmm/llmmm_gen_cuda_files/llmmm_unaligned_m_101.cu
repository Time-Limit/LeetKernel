#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_101_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<101> mm_instantiator;

public:
  UnalignedM_101_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_101_MMInstantiatorWrapper__;

}  // namespace LLMMM
