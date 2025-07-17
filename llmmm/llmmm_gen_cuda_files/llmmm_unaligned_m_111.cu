#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_111_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<111> mm_instantiator;

public:
  UnalignedM_111_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_111_MMInstantiatorWrapper__;

}  // namespace LLMMM
