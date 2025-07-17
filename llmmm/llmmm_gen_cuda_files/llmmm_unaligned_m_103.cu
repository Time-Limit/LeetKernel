#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_103_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<103> mm_instantiator;

public:
  UnalignedM_103_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_103_MMInstantiatorWrapper__;

}  // namespace LLMMM
