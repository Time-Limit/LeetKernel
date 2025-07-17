#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_5_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<5> mm_instantiator;

public:
  UnalignedM_5_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_5_MMInstantiatorWrapper__;

}  // namespace LLMMM
