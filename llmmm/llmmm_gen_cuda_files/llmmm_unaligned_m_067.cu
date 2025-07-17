#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_67_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<67> mm_instantiator;

public:
  UnalignedM_67_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_67_MMInstantiatorWrapper__;

}  // namespace LLMMM
