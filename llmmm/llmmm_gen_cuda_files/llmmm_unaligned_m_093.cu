#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_93_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<93> mm_instantiator;

public:
  UnalignedM_93_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_93_MMInstantiatorWrapper__;

}  // namespace LLMMM
