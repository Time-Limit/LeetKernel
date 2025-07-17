#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_4_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<4> mm_instantiator;

public:
  UnalignedM_4_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_4_MMInstantiatorWrapper__;

}  // namespace LLMMM
