#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_124_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<124> mm_instantiator;

public:
  UnalignedM_124_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_124_MMInstantiatorWrapper__;

}  // namespace LLMMM
