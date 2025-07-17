#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_44_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<44> mm_instantiator;

public:
  UnalignedM_44_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_44_MMInstantiatorWrapper__;

}  // namespace LLMMM
